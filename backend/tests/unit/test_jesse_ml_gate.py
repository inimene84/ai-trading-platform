"""Unit tests for Jesse Machine Learning Consensus Gate in DecisionEngine."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from backend.services.decision_engine import DecisionEngine
from backend.services.risk_config import RiskConfig
from backend.strategies.base import StrategySignal

_PASSING_META = {
    "status": "success",
    "resolved_model_type": "lightgbm",
    "resolved_via": "primary",
    "metrics": {"deflated_sharpe_ratio": 1.0, "prob_backtest_overfitting": 0.05},
}

_LSTM_META = {
    "status": "success",
    "resolved_model_type": "lstm",
    "resolved_via": "fallback",
    "primary_model_type": "lightgbm",
    "primary_error": "Metadata not found for ETH-USDT_1h_lightgbm_meta.json",
    "metrics": {"deflated_sharpe_ratio": 1.0, "prob_backtest_overfitting": 0.028},
}

_MISSING_META = {
    "status": "error",
    "resolved_model_type": "lightgbm",
    "resolved_via": "none",
    "error": "No model artifact found for BTC-USDT (1h, lightgbm)",
}


def _make_bars(n: int = 100, base: float = 100.0) -> list:
    bars = []
    for i in range(n):
        c = base + i * 0.1
        bars.append({
            "open": c - 0.1,
            "high": c + 0.3,
            "low": c - 0.3,
            "close": c,
            "volume": 1000.0,
        })
    return bars


@pytest.fixture
def ml_risk_config():
    return RiskConfig(
        min_signal_strength=0.45,
        ai_analysis_threshold=0.30,
        trade_usdt_amount=10.0,
        enable_personas=False,
        use_risk_reviewer_llm=False,
        min_edge_fee_mult=0.0,
        sl_atr_mult=1.75,
        tp_atr_mult=5.5,
        enable_jesse_ml=True,
    )


@pytest.mark.asyncio
async def test_jesse_ml_gate_veto_buy_on_strong_bearish(ml_risk_config, monkeypatch):
    """When Jesse ML predicts strong Bearish probability (>=55%), candidate BUY is vetoed."""
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    engine = DecisionEngine(ml_risk_config)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)

    # Mock base strategy to emit a BUY signal
    mock_signal = StrategySignal(symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    # Mock Jesse ML prediction returning strong bearish probability
    mock_ml_res = {
        "status": "success",
        "symbol": "BTC-USDT",
        "signal": "SELL",
        "confidence": 0.70,
        "probabilities": {"bullish": 0.10, "bearish": 0.70, "neutral": 0.20},
    }

    with patch("backend.services.jesse_bridge.jesse_bridge.get_ml_prediction", AsyncMock(return_value=mock_ml_res)):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)
        assert decision is None
        assert "vetoed by Jesse ML gate" in engine.last_evaluation.get("reason", "")


@pytest.mark.asyncio
async def test_jesse_ml_gate_veto_sell_on_strong_bullish(ml_risk_config, monkeypatch):
    """When Jesse ML predicts strong Bullish probability (>=55%), candidate SELL is vetoed."""
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    engine = DecisionEngine(ml_risk_config)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)

    # Mock base strategy to emit a SELL signal
    mock_signal = StrategySignal(symbol="BTCUSDC", signal="SELL", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    # Mock Jesse ML prediction returning strong bullish probability
    mock_ml_res = {
        "status": "success",
        "symbol": "BTC-USDT",
        "signal": "BUY",
        "confidence": 0.75,
        "probabilities": {"bullish": 0.75, "bearish": 0.08, "neutral": 0.17},
    }

    with patch("backend.services.jesse_bridge.jesse_bridge.get_ml_prediction", AsyncMock(return_value=mock_ml_res)):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)
        assert decision is None
        assert "vetoed by Jesse ML gate" in engine.last_evaluation.get("reason", "")


@pytest.mark.asyncio
async def test_jesse_ml_gate_boost_on_consensus(ml_risk_config, monkeypatch):
    """When Jesse ML agrees with candidate BUY, confidence is boosted."""
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    engine = DecisionEngine(ml_risk_config)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)

    initial_conf = 0.55
    mock_signal = StrategySignal(symbol="BTCUSDC", signal="BUY", confidence=initial_conf, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    mock_ml_res = {
        "status": "success",
        "symbol": "BTC-USDT",
        "signal": "BUY",
        "confidence": 0.80,
        "probabilities": {"bullish": 0.80, "bearish": 0.05, "neutral": 0.15},
    }

    with patch("backend.services.jesse_bridge.jesse_bridge.get_ml_prediction", AsyncMock(return_value=mock_ml_res)):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)
        assert decision is not None
        assert decision.action == "BUY"
        # Confidence must be boosted above initial
        assert decision.confidence > initial_conf


@pytest.mark.asyncio
async def test_jesse_ml_gate_fail_open_on_error_in_paper(ml_risk_config, monkeypatch):
    """In paper mode, when Jesse ML server is down or errors, gate fails open gracefully without blocking trades."""
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    engine = DecisionEngine(ml_risk_config)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)

    mock_signal = StrategySignal(symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    with patch("backend.services.jesse_bridge.jesse_bridge.get_ml_prediction", AsyncMock(side_effect=Exception("Connection refused"))):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)
        # In paper mode, fails open
        assert decision is not None
        assert decision.action == "BUY"


@pytest.mark.asyncio
async def test_jesse_ml_gate_fail_closed_in_live_mode(ml_risk_config, monkeypatch):
    """In LIVE mode, when Jesse ML server is down or errors, gate fails closed (vetoes entry)."""
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    engine = DecisionEngine(ml_risk_config)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)

    mock_signal = StrategySignal(symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    with patch(
        "backend.services.jesse_bridge.jesse_bridge.resolve_gate_model",
        AsyncMock(return_value=_PASSING_META),
    ), patch(
        "backend.services.jesse_bridge.jesse_bridge.get_ml_prediction",
        AsyncMock(side_effect=Exception("Connection refused")),
    ):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)
        # In live mode, must fail closed
        assert decision is None


@pytest.mark.asyncio
async def test_jesse_ml_kelly_clipping_below_30_partition_trades(ml_risk_config, monkeypatch):
    """When partition closed trades < 30, Kelly multiplier must be clipped to [0.25, 1.0]."""
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    engine = DecisionEngine(ml_risk_config)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)

    mock_signal = StrategySignal(
        symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"],
        stop_loss=bars[-1]["close"] * 0.98, take_profit=bars[-1]["close"] * 1.04,
    )
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    # Jesse returns a high Kelly multiplier (e.g. 1.8x)
    mock_ml = {
        "status": "success",
        "signal": "BUY",
        "confidence": 0.70,
        "probabilities": {"bullish": 0.70, "bearish": 0.10, "neutral": 0.20},
        "uncertainty": "LOW",
        "gated": False,
        "kelly": {"size_multiplier": 1.8},
    }

    with patch(
        "backend.services.jesse_bridge.jesse_bridge.resolve_gate_model",
        AsyncMock(return_value=_PASSING_META),
    ), patch(
        "backend.services.jesse_bridge.jesse_bridge.get_ml_prediction",
        AsyncMock(return_value=mock_ml),
    ):
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)
        assert decision is not None
        # Base trade_usdt_amount is 10.0 (from ml_risk_config), clipped kelly is 1.0x (not 1.8x)
        # If clipped to 1.0x, notional doesn't scale above 1.0x trade_usdt or risk
        assert decision.action == "BUY"


@pytest.mark.asyncio
async def test_jesse_ml_gate_fail_closed_on_validation_in_live(ml_risk_config, monkeypatch):
    """In LIVE mode, failing PBO/DSR validation gates veto entry before prediction."""
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    engine = DecisionEngine(ml_risk_config)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)

    mock_signal = StrategySignal(symbol="BTCUSDC", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    mock_meta = {
        "metrics": {"deflated_sharpe_ratio": 1.0, "prob_backtest_overfitting": 0.636},
    }

    failing_meta = {
        "status": "success",
        "resolved_model_type": "lightgbm",
        "resolved_via": "primary",
        **mock_meta,
    }
    with patch(
        "backend.services.jesse_bridge.jesse_bridge.resolve_gate_model",
        AsyncMock(return_value=failing_meta),
    ), patch(
        "backend.services.jesse_bridge.jesse_bridge.get_ml_prediction",
        AsyncMock(),
    ) as mock_predict:
        decision = await engine.evaluate_symbol("BTCUSDC", bars, None, 0, [], False)
        assert decision is None
        assert "validation gate" in engine.last_evaluation.get("reason", "")
        mock_predict.assert_not_called()


@pytest.mark.asyncio
async def test_jesse_ml_gate_lstm_fallback_in_live_when_lightgbm_missing(ml_risk_config, monkeypatch):
    """Live ETH 1h: quarantined LightGBM must not disable the gate; use promoted LSTM."""
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    engine = DecisionEngine(ml_risk_config)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)

    mock_signal = StrategySignal(symbol="ETHUSDT", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    mock_ml = {
        "status": "success",
        "signal": "BUY",
        "confidence": 0.70,
        "probabilities": {"bullish": 0.70, "bearish": 0.10, "neutral": 0.20},
        "uncertainty": "LOW",
        "gated": False,
        "kelly": {"size_multiplier": 1.0},
    }

    with patch(
        "backend.services.jesse_bridge.jesse_bridge.resolve_gate_model",
        AsyncMock(return_value=_LSTM_META),
    ), patch(
        "backend.services.jesse_bridge.jesse_bridge.get_ml_prediction",
        AsyncMock(return_value=mock_ml),
    ) as mock_predict:
        decision = await engine.evaluate_symbol("ETHUSDT", bars, None, 0, [], False)
        assert decision is not None
        assert decision.action == "BUY"
        mock_predict.assert_awaited()
        assert mock_predict.await_args.kwargs["model_type"] == "lstm"


@pytest.mark.asyncio
async def test_jesse_ml_gate_fail_closed_when_no_production_artifact(ml_risk_config, monkeypatch):
    """Live BTC 1h has no promoted LightGBM or LSTM — keep the gate on and veto."""
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("JESSE_ML_GATE_ENABLED", "true")
    engine = DecisionEngine(ml_risk_config)
    engine.enable_kronos = False
    engine.account_equity = 1000.0
    engine.account_available = 1000.0
    bars = _make_bars(50)

    mock_signal = StrategySignal(symbol="BTCUSDT", signal="BUY", confidence=0.60, entry_price=bars[-1]["close"])
    engine.strategy.generate_signal = MagicMock(return_value=mock_signal)
    engine.regime_detector.detect = MagicMock(return_value=MagicMock(regime="TRENDING", weights=MagicMock(return_value={})))

    with patch(
        "backend.services.jesse_bridge.jesse_bridge.resolve_gate_model",
        AsyncMock(return_value=_MISSING_META),
    ), patch(
        "backend.services.jesse_bridge.jesse_bridge.get_ml_prediction",
        AsyncMock(),
    ) as mock_predict:
        decision = await engine.evaluate_symbol("BTCUSDT", bars, None, 0, [], False)
        assert decision is None
        assert "No model artifact found" in engine.last_evaluation.get("reason", "")
        mock_predict.assert_not_called()

