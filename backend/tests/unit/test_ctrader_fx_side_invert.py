"""cTrader FX side invert: polarity flip + SL/TP geometry.

IC demo closed book (Sep 11–24, n=55) was systematically on the losing side
of the subsequent move. Invert is gated to cTrader forex only — Binance and
metals must keep their original side.
"""

import time
from unittest.mock import AsyncMock, patch

import pytest

from backend.services.multi_asset_bars import classify_symbol
from backend.services.signal_candidate_engine import (
    CandidateStatus,
    SignalCandidateEngine,
    TimingMode,
)
from backend.services.ctrader_service import CTraderService


def test_invert_buy_mirrors_sl_below_to_sl_above():
    flipped, sl, tp = SignalCandidateEngine.invert_side_and_protection(
        "BUY", 1.0850, 1.0820, 1.0910, digits=5,
    )
    assert flipped == "SELL"
    assert sl == pytest.approx(1.0880)
    assert tp == pytest.approx(1.0790)


def test_invert_sell_mirrors_sl_above_to_sl_below():
    flipped, sl, tp = SignalCandidateEngine.invert_side_and_protection(
        "SELL", 1.0850, 1.0880, 1.0790, digits=5,
    )
    assert flipped == "BUY"
    assert sl == pytest.approx(1.0820)
    assert tp == pytest.approx(1.0910)


def test_invert_long_alias_and_none_levels():
    flipped, sl, tp = SignalCandidateEngine.invert_side_and_protection(
        "LONG", 150.0, None, None, digits=3,
    )
    assert flipped == "SELL"
    assert sl is None
    assert tp is None


def test_post_invert_clamp_keeps_stop_on_loss_side():
    """Mirroring then clamp must keep SELL stop above entry and TP below."""
    entry = 1.0850
    flipped, sl, tp = SignalCandidateEngine.invert_side_and_protection(
        "BUY", entry, 1.0820, 1.0910, digits=5,
    )
    clamped_sl, clamped_tp = CTraderService.clamp_protective_prices(
        "EURUSD", entry, sl, tp, direction=flipped,
    )
    assert flipped == "SELL"
    assert clamped_sl > entry
    assert clamped_tp < entry
    assert abs(clamped_sl - entry) == pytest.approx(abs(1.0820 - entry))
    assert abs(clamped_tp - entry) == pytest.approx(abs(1.0910 - entry))


def test_post_invert_jpy_geometry_stays_valid():
    entry = 150.000
    flipped, sl, tp = SignalCandidateEngine.invert_side_and_protection(
        "BUY", entry, 149.500, 151.000, digits=3,
    )
    clamped_sl, clamped_tp = CTraderService.clamp_protective_prices(
        "USDJPY", entry, sl, tp, direction=flipped,
    )
    validated = CTraderService.validate_protective_geometry(
        "USDJPY", entry, sl, tp, clamped_sl, clamped_tp, direction=flipped,
    )
    assert flipped == "SELL"
    assert validated is not None
    out_sl, out_tp = validated
    assert out_sl > entry
    assert out_tp < entry


def test_should_invert_only_ctrader_forex():
    engine = SignalCandidateEngine()
    assert classify_symbol("EURUSD") == "forex"
    assert classify_symbol("XAUUSD") == "metal"
    assert classify_symbol("BTCUSDT") == "crypto"
    with patch.object(engine, "_ctrader_invert_side_enabled", return_value=True):
        assert engine._should_invert_ctrader_fx_side("ctrader", "EURUSD") is True
        assert engine._should_invert_ctrader_fx_side("ctrader", "USDJPY") is True
        assert engine._should_invert_ctrader_fx_side("ctrader", "XAUUSD") is False
        assert engine._should_invert_ctrader_fx_side("binance_futures", "EURUSD") is False
        assert engine._should_invert_ctrader_fx_side("binance_futures", "BTCUSDT") is False
    with patch.object(engine, "_ctrader_invert_side_enabled", return_value=False):
        assert engine._should_invert_ctrader_fx_side("ctrader", "EURUSD") is False


def test_apply_invert_is_idempotent_and_annotates():
    engine = SignalCandidateEngine()
    signal = {
        "direction": "BUY",
        "entry_price": 1.0850,
        "stop_loss": 1.0820,
        "take_profit": 1.0910,
        "reason": "M5 BUY momentum confirmed.",
    }
    once = engine.apply_ctrader_fx_side_invert("EURUSD", signal)
    assert once is signal
    assert signal["direction"] == "SELL"
    assert signal["original_direction"] == "BUY"
    assert signal["side_inverted"] is True
    assert signal["stop_loss"] == pytest.approx(1.0880)
    assert signal["take_profit"] == pytest.approx(1.0790)
    assert "BUY→SELL" in signal["reason"]

    twice = engine.apply_ctrader_fx_side_invert("EURUSD", signal)
    assert twice is signal
    assert signal["direction"] == "SELL"
    assert signal["stop_loss"] == pytest.approx(1.0880)


def test_flag_off_leaves_candidate_side_unchanged():
    engine = SignalCandidateEngine()
    cand = {
        "broker": "ctrader",
        "symbol": "EURUSD",
        "direction": "BUY",
        "entry_price": 1.0850,
        "stop_loss": 1.0820,
        "take_profit": 1.0910,
        "sizing": {"lots": 0.01},
    }
    with patch.object(engine, "_ctrader_invert_side_enabled", return_value=False):
        assert engine._ensure_ctrader_fx_side_invert(cand) is True
    assert cand["direction"] == "BUY"
    assert cand["stop_loss"] == 1.0820
    assert cand.get("side_inverted") is None


def _ready_fx_candidate(cid: str, now_ts: int, *, direction: str = "BUY") -> dict:
    sl = 1.0820 if direction == "BUY" else 1.0880
    tp = 1.0910 if direction == "BUY" else 1.0790
    return {
        "id": cid,
        "symbol": "EURUSD",
        "broker": "ctrader",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": direction,
        "entry_price": 1.0850,
        "stop_loss": sl,
        "take_profit": tp,
        "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.01, "quantity": 0.01, "risk_usd": 5.0},
    }


@pytest.mark.asyncio
async def test_execute_candidate_sends_inverted_side_and_mirrored_protection():
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    now_ts = int(time.time())
    engine.candidates["fx-1"] = _ready_fx_candidate("fx-1", now_ts, direction="BUY")

    with patch("backend.services.sentry_state.is_trading_allowed", return_value=True), \
         patch.object(engine, "_ctrader_execution_slot_available", return_value=True), \
         patch.object(engine, "_open_ctrader_symbols", return_value=set()), \
         patch.object(engine, "_same_base_slots_available", return_value=True), \
         patch.object(engine, "_currency_exposure_slots_available", return_value=True), \
         patch.object(engine, "_has_open_position", return_value=False), \
         patch.object(engine, "_portfolio_risk_breach", return_value=None), \
         patch.object(engine, "_calculate_size", return_value={"lots": 0.01, "quantity": 0.01, "risk_usd": 5.0}), \
         patch("backend.services.signal_candidate_engine.live_ctrader_orders_allowed", return_value=False), \
         patch("backend.services.signal_candidate_engine.ctrader_service.get_spread_pips", return_value=0.4), \
         patch(
             "backend.services.signal_candidate_engine.ctrader_service.place_order",
             return_value={"status": "sent", "order_id": "oid-1", "direction": "SELL"},
         ) as mock_place, \
         patch("backend.services.signal_candidate_engine.persist_ctrader_execution", return_value=1) as mock_persist:
        res = await engine.execute_candidate("fx-1", force=True)

    assert res["success"] is True
    assert mock_place.call_args.kwargs["direction"] == "SELL"
    assert mock_place.call_args.kwargs["stop_loss"] == pytest.approx(1.0880)
    assert mock_place.call_args.kwargs["take_profit"] == pytest.approx(1.0790)
    assert mock_persist.call_args.kwargs["direction"] == "SELL"
    assert mock_persist.call_args.kwargs["stop_loss"] == pytest.approx(1.0880)
    cand = engine.candidates["fx-1"]
    assert cand["direction"] == "SELL"
    assert cand["original_direction"] == "BUY"
    assert cand["side_inverted"] is True


@pytest.mark.asyncio
async def test_execute_candidate_does_not_invert_binance():
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    now_ts = int(time.time())
    engine.candidates["b-1"] = {
        "id": "b-1",
        "symbol": "BTCUSDT",
        "broker": "binance_futures",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 100000,
        "stop_loss": 99000,
        "take_profit": 102000,
        "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.01, "quantity": 0.01, "risk_usd": 50.0},
    }
    mock_order = type("R", (), {"success": True, "order_id": "b", "message": "ok"})()
    with patch.dict(engine.execution_config, {"forex_only": False, "max_portfolio_risk_pct": 0}), \
         patch("backend.services.sentry_state.is_trading_allowed", return_value=True), \
         patch.object(engine, "_try_open_binance_positions", return_value=[]), \
         patch.object(engine, "_resolve_mark_price", new_callable=AsyncMock, return_value=100000.0), \
         patch("backend.services.signal_candidate_engine.UnifiedTrading") as ut:
        ut.return_value.list_sessions.return_value = [
            {"id": "binance_futures_paper", "broker": "binance_futures", "mode": "paper"}
        ]
        ut.return_value.place_order.return_value = mock_order
        res = await engine.execute_candidate("b-1", force=True)

    assert res["success"] is True
    order = ut.return_value.place_order.call_args.args[0]
    assert order.side.value.upper() == "BUY"
    assert engine.candidates["b-1"]["direction"] == "BUY"
    assert engine.candidates["b-1"].get("side_inverted") in (None, False)


@pytest.mark.asyncio
async def test_execute_candidate_does_not_invert_metal():
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    now_ts = int(time.time())
    engine.candidates["xau-1"] = {
        "id": "xau-1",
        "symbol": "XAUUSD",
        "broker": "ctrader",
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 2500.0,
        "stop_loss": 2490.0,
        "take_profit": 2520.0,
        "timing_mode": TimingMode.BAR_CLOSE,
        "status": CandidateStatus.READY,
        "earliest_exec_at": now_ts - 5,
        "latest_exec_at": now_ts + 600,
        "sizing": {"lots": 0.01, "quantity": 0.01, "risk_usd": 5.0},
    }
    with patch("backend.services.sentry_state.is_trading_allowed", return_value=True), \
         patch.dict(engine.execution_config, {"include_metals": True, "max_portfolio_risk_pct": 0}), \
         patch.object(engine, "_ctrader_execution_slot_available", return_value=True), \
         patch.object(engine, "_open_ctrader_symbols", return_value=set()), \
         patch.object(engine, "_same_base_slots_available", return_value=True), \
         patch.object(engine, "_currency_exposure_slots_available", return_value=True), \
         patch.object(engine, "_has_open_position", return_value=False), \
         patch("backend.services.signal_candidate_engine.live_ctrader_orders_allowed", return_value=False), \
         patch("backend.services.signal_candidate_engine.ctrader_service.get_spread_pips", return_value=0.4), \
         patch(
             "backend.services.signal_candidate_engine.ctrader_service.place_order",
             return_value={"status": "sent", "order_id": "xau", "direction": "BUY"},
         ) as mock_place, \
         patch("backend.services.signal_candidate_engine.persist_ctrader_execution", return_value=1):
        res = await engine.execute_candidate("xau-1", force=True)

    assert res["success"] is True
    assert mock_place.call_args.kwargs["direction"] == "BUY"
    assert mock_place.call_args.kwargs["stop_loss"] == 2490.0
    assert engine.candidates["xau-1"]["direction"] == "BUY"


@pytest.mark.asyncio
async def test_scan_markets_stores_inverted_fx_side():
    engine = SignalCandidateEngine()
    engine.candidates.clear()
    raw = {
        "strategy": "MOMENTUM_TREND_PULSE",
        "direction": "BUY",
        "entry_price": 1.0850,
        "stop_loss": 1.0820,
        "take_profit": 1.0910,
        "timing_mode": TimingMode.POST_REACTION,
        "confidence": 0.82,
        "reason": "M5 BUY momentum confirmed.",
    }
    bars = [{"close": 1.085, "high": 1.086, "low": 1.084, "volume": 1} for _ in range(25)]
    with patch.object(CTraderService, "get_trendbars", return_value=bars), \
         patch.object(engine, "_attach_stop_atr", new_callable=AsyncMock), \
         patch.object(engine, "_evaluate_momentum", side_effect=lambda *a, **k: dict(raw)), \
         patch.object(engine, "_evaluate_fade", return_value=None), \
         patch.object(engine, "_evaluate_straddle", return_value=None), \
         patch.object(engine, "_evaluate_slingshot", return_value=None), \
         patch.object(engine, "_fx_gate_mode", return_value="off"), \
         patch.object(engine, "_has_open_position", return_value=False), \
         patch.object(engine, "_portfolio_risk_breach", return_value=None), \
         patch("backend.services.signal_candidate_engine.is_venue_open", return_value=True):
        created = await engine.scan_markets(universe=["EURUSD"], timeframe="M5")

    assert len(created) == 1
    cand = created[0]
    assert cand["direction"] == "SELL"
    assert cand["original_direction"] == "BUY"
    assert cand["side_inverted"] is True
    assert cand["stop_loss"] > cand["entry_price"]
    assert cand["take_profit"] < cand["entry_price"]
