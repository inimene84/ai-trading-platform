"""Fail-closed audit recreation: BE ratchet, telemetry veto, /jesse/sync 409."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from backend.main import app

from backend.ml.promotion_gates import GateResult
from backend.ml.promotion_service import PromotionState
from backend.services.decision_engine import DecisionEngine
from backend.services.jesse_bridge import JesseBridgeService
from backend.services.risk_config import RiskConfig
from backend.services.trading_loop_helpers import (
    PartialTPManager,
    TrailingStopManager,
    _ratchet_stop_to_be_fees,
)
from backend.strategies.base import StrategySignal
from backend.tests.unit.test_jesse_ml_gate import _make_bars


def test_ratchet_stop_to_be_fees_long_tightens_only():
    cfg = RiskConfig(taker_fee_rate=0.0004, slippage_rate=0.0002)
    trade = SimpleNamespace(
        symbol="ETHUSDT", direction="BUY", entry_price=100.0, stop_loss=98.0,
    )
    with patch.object(TrailingStopManager, "_sync_exchange_stop") as sync:
        moved = _ratchet_stop_to_be_fees(trade, cfg, broker="broker", mark=104.0)
    assert moved is True
    assert trade.stop_loss == pytest.approx(100.12)
    sync.assert_called_once()
    assert sync.call_args[0][1] == pytest.approx(100.12)

    with patch.object(TrailingStopManager, "_sync_exchange_stop") as sync:
        moved_again = _ratchet_stop_to_be_fees(trade, cfg, broker="broker", mark=104.0)
    assert moved_again is False
    sync.assert_not_called()


def test_ratchet_stop_to_be_fees_short_and_refuses_widen():
    cfg = RiskConfig(taker_fee_rate=0.0004, slippage_rate=0.0002)
    trade = SimpleNamespace(
        symbol="ETHUSDT", direction="SELL", entry_price=100.0, stop_loss=103.0,
    )
    with patch.object(TrailingStopManager, "_sync_exchange_stop") as sync:
        assert _ratchet_stop_to_be_fees(trade, cfg, broker="broker", mark=96.0) is True
    assert trade.stop_loss == pytest.approx(99.88)
    sync.assert_called_once()

    tighter = SimpleNamespace(
        symbol="ETHUSDT", direction="SELL", entry_price=100.0, stop_loss=99.50,
    )
    with patch.object(TrailingStopManager, "_sync_exchange_stop") as sync:
        assert _ratchet_stop_to_be_fees(tighter, cfg, broker="broker", mark=96.0) is False
    assert tighter.stop_loss == 99.50
    sync.assert_not_called()


def test_ratchet_refuses_wrong_side_of_mark():
    cfg = RiskConfig(taker_fee_rate=0.0004, slippage_rate=0.0002)
    trade = SimpleNamespace(
        symbol="ETHUSDT", direction="BUY", entry_price=100.0, stop_loss=98.0,
    )
    with patch.object(TrailingStopManager, "_sync_exchange_stop") as sync:
        # mark still at entry — BE+fees would sit above mark
        assert _ratchet_stop_to_be_fees(trade, cfg, broker="broker", mark=100.05) is False
    assert trade.stop_loss == 98.0
    sync.assert_not_called()


def test_paper_partial_tp_ratchets_after_fill():
    cfg = RiskConfig(
        partial_tp_enabled=True,
        partial_tp_atr_mult=0.5,
        partial_tp_close_pct=0.5,
        taker_fee_rate=0.0004,
        slippage_rate=0.0002,
    )
    bars = []
    for _ in range(20):
        bars.append({"open": 99.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0})
    bars[-1]["close"] = 104.0
    trade = SimpleNamespace(
        id=1, symbol="ETHUSDT", direction="BUY", entry_price=100.0,
        quantity=2.0, notes=None, status="open", stop_loss=98.0,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [trade]
    fill = MagicMock(success=True, filled_price=104.0, message="ok")
    with patch("backend.services.trading_loop_helpers.UnifiedTrading") as ut_cls, \
         patch.object(TrailingStopManager, "_sync_exchange_stop") as sync:
        ut_cls.return_value.place_order.return_value = fill
        PartialTPManager.apply_partial_tp(db, "ETHUSDT", bars, cfg, "Test")
    assert "PARTIAL_TP_DONE" in (trade.notes or "")
    assert trade.stop_loss == pytest.approx(100.12)
    sync.assert_called_once()


@pytest.mark.asyncio
async def test_promoted_live_missing_telemetry_vetoes(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    cfg = RiskConfig(
        min_signal_strength=0.45,
        ai_analysis_threshold=0.30,
        trade_usdt_amount=10.0,
        enable_personas=False,
        use_risk_reviewer_llm=False,
        min_edge_fee_mult=0.0,
        enable_jesse_ml=True,
    )
    engine = DecisionEngine(cfg)
    engine.enable_kronos = False
    engine.promotion_state = PromotionState(
        result=GateResult(verdict="PROMOTE", reason="PASS", failed_gate=None),
    )
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)
    mock_signal = StrategySignal(
        symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"],
    )
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(
        return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={}))
    )
    mock_ml = {
        "status": "success",
        "signal": "BUY",
        "confidence": 0.70,
        "probabilities": {"bullish": 0.70, "bearish": 0.10, "neutral": 0.20},
        "uncertainty": "LOW",
        "gated": False,
    }
    with patch(
        "backend.services.decision_engine.jesse_bridge.get_ml_prediction",
        AsyncMock(return_value=mock_ml),
    ):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)
    assert decision is None
    assert "missing live telemetry" in engine.last_evaluation.get("reason", "")


@pytest.mark.asyncio
async def test_promoted_paper_missing_telemetry_fail_open(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    cfg = RiskConfig(
        min_signal_strength=0.45,
        ai_analysis_threshold=0.30,
        trade_usdt_amount=10.0,
        enable_personas=False,
        use_risk_reviewer_llm=False,
        min_edge_fee_mult=0.0,
        enable_jesse_ml=True,
    )
    engine = DecisionEngine(cfg)
    engine.enable_kronos = False
    engine.promotion_state = PromotionState(
        result=GateResult(verdict="PROMOTE", reason="PASS", failed_gate=None),
    )
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)
    mock_signal = StrategySignal(
        symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"],
    )
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(
        return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={}))
    )
    mock_ml = {
        "status": "success",
        "signal": "BUY",
        "confidence": 0.70,
        "probabilities": {"bullish": 0.70, "bearish": 0.10, "neutral": 0.20},
        "uncertainty": "LOW",
        "gated": False,
    }
    with patch(
        "backend.services.decision_engine.jesse_bridge.get_ml_prediction",
        AsyncMock(return_value=mock_ml),
    ):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)
    assert decision is not None
    assert decision.action == "BUY"


def test_jesse_sync_reject_restores_env_byte_for_byte(tmp_path, monkeypatch):
    dummy_env = tmp_path / ".env"
    original = b"SL_ATR_MULT=1.75\nTP_ATR_MULT=5.5\nTRAIL_ACTIVATION_ATR=2.0\nTRAIL_ATR_MULT=0.8\n# keep me\n"
    dummy_env.write_bytes(original)
    monkeypatch.setenv("ENV_FILE_PATH", str(dummy_env))
    monkeypatch.setenv("JESSE_SYNC_TO_LIVE", "true")
    monkeypatch.setenv("SL_ATR_MULT", "1.75")
    monkeypatch.setenv("TP_ATR_MULT", "5.5")

    rejected = PromotionState(
        result=GateResult(
            verdict="REJECT",
            reason="GEOMETRY_LIVE_LOCK: training geometry locked fields != live RiskConfig",
            failed_gate="GEOMETRY_LIVE_LOCK",
        )
    )
    service = JesseBridgeService()
    with patch("backend.services.jesse_bridge.resolve_promotion", return_value=rejected):
        result = service.sync_strategy_to_risk_config(
            sl_atr_mult=3.0, tp_atr_mult=6.0, trail_activation_atr=2.5, trail_atr_mult=1.0,
        )
    # G2: REJECT never reaches the .env writer.
    assert result["status"] == "blocked"
    assert result["synced"] is False
    assert result["live_sync_allowed"] is False
    assert dummy_env.read_bytes() == original
    assert os.environ.get("SL_ATR_MULT") == "1.75"
    assert os.environ.get("TP_ATR_MULT") == "5.5"


def test_jesse_sync_post_g2_reject_restores_env(tmp_path, monkeypatch):
    """If G2 was already PROMOTE, a later REJECT still restores .env."""
    dummy_env = tmp_path / ".env"
    original = b"SL_ATR_MULT=1.75\nTP_ATR_MULT=5.5\nTRAIL_ACTIVATION_ATR=2.0\nTRAIL_ATR_MULT=0.8\n# keep me\n"
    dummy_env.write_bytes(original)
    monkeypatch.setenv("ENV_FILE_PATH", str(dummy_env))
    monkeypatch.setenv("JESSE_SYNC_TO_LIVE", "true")
    monkeypatch.setenv("SL_ATR_MULT", "1.75")
    monkeypatch.setenv("TP_ATR_MULT", "5.5")
    monkeypatch.setattr(
        "backend.services.jesse_bridge.live_sync_contract",
        lambda risk_config=None: {
            "jesse_sync_to_live": True,
            "promotion_gates_required": True,
            "promoted": True,
            "verdict": "PROMOTE",
            "live_sync_allowed": True,
            "g2_ready": True,
        },
    )
    rejected = PromotionState(
        result=GateResult(
            verdict="REJECT",
            reason="GEOMETRY_LIVE_LOCK: training geometry locked fields != live RiskConfig",
            failed_gate="GEOMETRY_LIVE_LOCK",
        )
    )
    service = JesseBridgeService()
    with patch("backend.services.jesse_bridge.resolve_promotion", return_value=rejected):
        result = service.sync_strategy_to_risk_config(
            sl_atr_mult=3.0, tp_atr_mult=6.0, trail_activation_atr=2.5, trail_atr_mult=1.0,
        )
    assert result["status"] == "rejected"
    assert result["synced"] is False
    assert dummy_env.read_bytes() == original
    assert os.environ.get("SL_ATR_MULT") == "1.75"


def test_jesse_sync_exception_restores_env(tmp_path, monkeypatch):
    dummy_env = tmp_path / ".env"
    original = b"SL_ATR_MULT=1.75\nTP_ATR_MULT=5.5\n"
    dummy_env.write_bytes(original)
    monkeypatch.setenv("ENV_FILE_PATH", str(dummy_env))
    monkeypatch.setenv("JESSE_SYNC_TO_LIVE", "true")
    monkeypatch.setenv("SL_ATR_MULT", "1.75")
    monkeypatch.setattr(
        "backend.services.jesse_bridge.live_sync_contract",
        lambda risk_config=None: {
            "jesse_sync_to_live": True,
            "promotion_gates_required": True,
            "promoted": True,
            "verdict": "PROMOTE",
            "live_sync_allowed": True,
            "g2_ready": True,
        },
    )
    service = JesseBridgeService()
    with patch(
        "backend.services.jesse_bridge.resolve_promotion",
        side_effect=RuntimeError("artifact unreadable"),
    ):
        result = service.sync_strategy_to_risk_config(sl_atr_mult=3.0, tp_atr_mult=6.0)
    assert result["status"] == "rejected"
    assert dummy_env.read_bytes() == original
    assert os.environ.get("SL_ATR_MULT") == "1.75"


def test_jesse_sync_route_returns_409_on_reject(monkeypatch, tmp_path):
    dummy_env = tmp_path / ".env"
    dummy_env.write_text("SL_ATR_MULT=1.75\nTP_ATR_MULT=5.5\n")
    monkeypatch.setenv("ENV_FILE_PATH", str(dummy_env))
    monkeypatch.setenv("JESSE_SYNC_TO_LIVE", "true")
    api_key = os.getenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setenv("ADMIN_API_KEY", api_key)
    rejected = {
        "status": "rejected",
        "synced": False,
        "verdict": "REJECT",
        "reason": "do not promote B200 rejects",
        "failed_gate": "DSR",
        "http_status": 409,
    }
    with patch(
        "backend.routes.jesse.jesse_bridge.sync_strategy_to_risk_config",
        return_value=rejected,
    ):
        client = TestClient(app)
        response = client.post(
            "/api/jesse/sync",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={"sl_atr_mult": 3.0, "tp_atr_mult": 6.0},
        )
    assert response.status_code == 409
    assert response.json()["detail"]["status"] == "rejected"
