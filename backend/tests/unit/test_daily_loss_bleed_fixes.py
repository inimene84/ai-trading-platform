"""Tests for the daily-loss-bleed fixes: trail mark-cap, live partial TP,
captured-move min-edge, and Kronos FLIP→VETO.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.services.decision_engine import DecisionEngine
from backend.services.kronos_gate import apply_kronos_gate
from backend.services.risk_config import RiskConfig
from backend.services.trading_loop import TradingLoopService
from backend.services.trading_loop_helpers import PartialTPManager, TrailingStopManager


def _bars(n: int = 30, base: float = 100.0) -> list:
    bars = []
    for i in range(n):
        c = base
        bars.append(
            {
                "open": c - 0.1,
                "high": c + 1.0,
                "low": c - 1.0,
                "close": c,
                "volume": 1000.0,
            }
        )
    return bars


# ── Trailing stop mark-cap ──────────────────────────────────────────────────


def test_trail_does_not_clamp_stop_to_mark():
    """hw - trail_dist must be used even when that is below current price.

    Previously `min(candidate, current_price)` parked the stop on mark and
    scratched winners on the next tick.
    """
    cfg = RiskConfig(
        trailing_stop_enabled=True,
        native_trailing_enabled=False,
        step_trail_enabled=False,
        trail_activation_atr=1.0,
        trail_atr_mult=0.5,
    )
    bars = _bars(20, base=100.0)
    # Wide range so ATR is ~2.0; activation = 2.0, trail_dist = 1.0
    for b in bars:
        b["high"] = b["close"] + 1.0
        b["low"] = b["close"] - 1.0
    bars[-1]["close"] = 105.0
    bars[-1]["high"] = 106.0

    trade = MagicMock()
    trade.id = 1
    trade.symbol = "ETHUSDT"
    trade.direction = "BUY"
    trade.entry_price = 100.0
    trade.stop_loss = 98.0
    trade.quantity = 1.0

    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [trade]
    high_water: dict = {}

    with patch.object(TrailingStopManager, "_sync_exchange_stop"):
        TrailingStopManager.apply_trailing_stop(
            db, "ETHUSDT", bars, high_water, cfg, MagicMock()
        )

    # Stop must be below mark (not clamped to 105).
    assert trade.stop_loss < bars[-1]["close"]
    assert trade.stop_loss > trade.entry_price  # locked profit


def test_trail_skips_wrong_side_candidate():
    """If hw - trail_dist sits at/above mark (pullback), leave the stop unchanged."""
    cfg = RiskConfig(
        trailing_stop_enabled=True,
        native_trailing_enabled=False,
        step_trail_enabled=False,
        trail_activation_atr=0.5,
        trail_atr_mult=0.5,
    )
    bars = _bars(20, base=100.0)
    for b in bars:
        b["high"] = 101.0
        b["low"] = 99.0
        b["close"] = 100.0
    # Pullback: mark at 105 while high-water was 110 → candidate 110-trail > 105
    bars[-1]["close"] = 105.0
    bars[-1]["high"] = 105.5

    trade = MagicMock()
    trade.id = 2
    trade.symbol = "ETHUSDT"
    trade.direction = "BUY"
    trade.entry_price = 100.0
    trade.stop_loss = 98.0
    trade.quantity = 1.0

    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [trade]
    # Pre-seed high water well above current mark
    high_water = {2: 110.0}

    with patch.object(TrailingStopManager, "_sync_exchange_stop") as sync:
        TrailingStopManager.apply_trailing_stop(
            db, "ETHUSDT", bars, high_water, cfg, MagicMock()
        )
        # ATR ≈ 2, trail_dist ≈ 1 → candidate = 110 - 1 = 109 >= 105 → skipped
        sync.assert_not_called()
    assert trade.stop_loss == 98.0


# ── Live partial TP ─────────────────────────────────────────────────────────


def test_check_sl_tp_calls_live_partial_on_live_binance():
    loop = TradingLoopService()
    db = MagicMock()
    bars = [{"close": 100.0}]

    with patch.object(loop, "_ensure_exchange_protection"), \
         patch.object(loop, "_apply_trailing_stop"), \
         patch.object(loop, "_apply_partial_tp_live") as live_ptp, \
         patch.object(loop, "_apply_partial_tp") as paper_ptp, \
         patch.object(loop, "_is_live_binance", return_value=True):
        loop._check_sl_tp(db, "BTCUSDT", bars)
        live_ptp.assert_called_once()
        paper_ptp.assert_not_called()


def test_live_partial_tp_one_reduce_only_from_exchange_qty():
    cfg = RiskConfig(
        partial_tp_enabled=True,
        partial_tp_atr_mult=0.5,
        partial_tp_close_pct=0.5,
    )
    bars = _bars(20, base=100.0)
    for b in bars:
        b["high"] = 101.0
        b["low"] = 99.0
    bars[-1]["close"] = 104.0  # well into profit vs entry 100

    t1 = SimpleNamespace(
        id=1, symbol="ETHUSDT", direction="BUY", entry_price=100.0,
        quantity=1.0, notes=None, status="open", stop_loss=98.0,
    )
    t2 = SimpleNamespace(
        id=2, symbol="ETHUSDT", direction="BUY", entry_price=100.0,
        quantity=1.0, notes=None, status="open", stop_loss=98.0,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [t1, t2]

    broker = MagicMock()
    broker._to_futures_symbol.return_value = "ETHUSDT"
    broker._live_position_qty.side_effect = [2.0, 1.0]  # before close, after fill
    broker._round_qty.side_effect = lambda sym, qty, **kwargs: round(qty, 3)

    fill = MagicMock()
    fill.success = True
    fill.filled_price = 104.0
    fill.message = "ok"

    with patch("backend.services.trading_loop_helpers.UnifiedTrading") as ut_cls:
        ut_cls.return_value.place_order.return_value = fill
        PartialTPManager.apply_partial_tp_live(db, "ETHUSDT", bars, cfg, broker)

        ut_cls.return_value.place_order.assert_called_once()
        order = ut_cls.return_value.place_order.call_args[0][0]
        assert order.reduce_only is True
        assert abs(order.quantity - 1.0) < 1e-9  # 50% of live 2.0, not per-row 0.5+0.5 over-close

    # DB rows rescaled to remaining live qty (1.0 total across 2 rows → 0.5 each)
    assert abs(t1.quantity + t2.quantity - 1.0) < 1e-6
    assert "PARTIAL_TP_DONE" in (t1.notes or "")
    assert "PARTIAL_TP_DONE" in (t2.notes or "")
    be_fees = 100.0 * cfg.roundtrip_cost_rate
    assert t1.stop_loss == pytest.approx(100.0 + be_fees)
    assert t2.stop_loss == pytest.approx(100.0 + be_fees)


def test_live_partial_tp_idempotent_second_call():
    cfg = RiskConfig(partial_tp_enabled=True, partial_tp_atr_mult=0.1, partial_tp_close_pct=0.5)
    bars = _bars(20, base=100.0)
    bars[-1]["close"] = 110.0

    trade = SimpleNamespace(
        id=1, symbol="ETHUSDT", direction="BUY", entry_price=100.0,
        quantity=1.0, notes=" | PARTIAL_TP_DONE: already", status="open",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [trade]
    broker = MagicMock()

    with patch("backend.services.trading_loop_helpers.UnifiedTrading") as ut_cls:
        PartialTPManager.apply_partial_tp_live(db, "ETHUSDT", bars, cfg, broker)
        ut_cls.assert_not_called()
        broker._live_position_qty.assert_not_called()


# ── Min-edge captured move ──────────────────────────────────────────────────


def test_min_edge_uses_trail_capture_not_full_tp():
    """Large TP but tiny trail lock must fail the fee gate."""
    cfg = RiskConfig(
        min_edge_fee_mult=2.5,
        taker_fee_rate=0.0004,
        slippage_rate=0.0002,
        trailing_stop_enabled=True,
        trail_activation_atr=1.0,
        trail_atr_mult=0.9,  # captured = 0.1 ATR — tiny
        partial_tp_enabled=False,
        tp_atr_mult=10.0,    # theoretical TP huge
        sl_atr_mult=1.0,
        equity_sizing_enabled=False,
        trade_usdt_amount=100.0,
    )
    engine = DecisionEngine(cfg)
    # Flat bars → ATR from high-low ≈ 2.0
    bars = _bars(30, base=100.0)
    for b in bars:
        b["high"] = 101.0
        b["low"] = 99.0

    # expected_move = 0.1 * atr ≈ 0.2; gross on $100 notional qty=1 → $0.20
    # roundtrip = 0.0012 * 100 = 0.12; required = 2.5 * 0.12 = 0.30 → fail
    assert engine._passes_min_edge("ETHUSDT", 100.0, 120.0, 1.0, bars) is False


def test_min_edge_blends_partial_tp_capture():
    """With partial TP on, captured ATR is 50% × 1.0 + 50% × trail lock."""
    bars = _bars(30, base=100.0)
    for b in bars:
        b["high"] = 101.0
        b["low"] = 99.0
    # ATR ≈ 2.0. Blend 1.1 → $2.20; trail-only 1.2 → $2.40.
    # min_edge_fee_mult=19 → required $2.28, so blend fails and trail-only would pass.
    blended = RiskConfig(
        min_edge_fee_mult=19.0,
        taker_fee_rate=0.0004,
        slippage_rate=0.0002,
        trailing_stop_enabled=True,
        trail_activation_atr=2.0,
        trail_atr_mult=0.8,
        partial_tp_enabled=True,
        partial_tp_atr_mult=1.0,
        partial_tp_close_pct=0.5,
    )
    trail_only = RiskConfig(
        min_edge_fee_mult=19.0,
        taker_fee_rate=0.0004,
        slippage_rate=0.0002,
        trailing_stop_enabled=True,
        trail_activation_atr=2.0,
        trail_atr_mult=0.8,
        partial_tp_enabled=False,
    )
    assert DecisionEngine(blended)._passes_min_edge("ETHUSDT", 100.0, 120.0, 1.0, bars) is False
    assert DecisionEngine(trail_only)._passes_min_edge("ETHUSDT", 100.0, 120.0, 1.0, bars) is True


def test_min_edge_full_tp_when_trailing_disabled():
    cfg = RiskConfig(
        min_edge_fee_mult=2.5,
        taker_fee_rate=0.0004,
        slippage_rate=0.0002,
        trailing_stop_enabled=False,
    )
    engine = DecisionEngine(cfg)
    bars = _bars(30, base=100.0)
    # tp_distance=5, qty=1 → gross $5; roundtrip $0.12; required $0.30 → pass
    assert engine._passes_min_edge("ETHUSDT", 100.0, 105.0, 1.0, bars) is True
    # tiny TP fails
    assert engine._passes_min_edge("ETHUSDT", 100.0, 100.05, 1.0, bars) is False


# ── Kronos FLIP → VETO ──────────────────────────────────────────────────────


def test_kronos_strong_opposition_vetoes_not_flips():
    # Active mode: hard veto must neutralize (never FLIP).
    result = apply_kronos_gate(
        strategy_signal="BUY",
        strategy_confidence=0.8,
        kronos_result={
            "signal": "SELL",
            "confidence": 0.75,
            "predicted_change_pct": -1.5,
            "cum_change_5_pct": -1.5,
        },
        shadow_mode=False,
        symbol="ETHUSDT",
    )
    assert result.action == "veto"
    assert result.final_signal == "NEUTRAL"
    assert result.confidence == 0.0
    assert result.is_shadow_veto is False


def test_kronos_boost_still_works():
    result = apply_kronos_gate(
        strategy_signal="BUY",
        strategy_confidence=0.7,
        kronos_result={
            "signal": "BUY",
            "confidence": 0.8,
            "predicted_change_pct": 1.2,
            "cum_change_5_pct": 1.2,
        },
        shadow_mode=False,
        symbol="ETHUSDT",
    )
    assert result.action == "boost"
    assert result.final_signal == "BUY"
    assert result.confidence > 0.7
