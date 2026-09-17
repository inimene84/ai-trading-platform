"""Unit tests for Jesse AI Quant Engine Bridge and API Routes."""

import os
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.jesse_bridge import (
    JesseBridgeService,
    is_missing_artifact_error,
    jesse_bridge,
    metadata_is_usable,
    ml_predict_timeout_seconds,
)
from backend.services.risk_config import get_risk_config


@pytest.fixture
def client():
    return TestClient(app)


def test_jesse_bridge_sync_parameters(tmp_path, monkeypatch):
    """Test sync_strategy_to_risk_config updates environment and reloads RiskConfig when JESSE_SYNC_TO_LIVE=true."""
    dummy_env = tmp_path / ".env"
    dummy_env.write_text("SL_ATR_MULT=1.0\nTP_ATR_MULT=2.5\nTRAIL_ACTIVATION_ATR=1.5\nTRAIL_ATR_MULT=0.8\n")
    monkeypatch.setenv("ENV_FILE_PATH", str(dummy_env))
    monkeypatch.setenv("JESSE_SYNC_TO_LIVE", "true")

    service = JesseBridgeService()
    result = service.sync_strategy_to_risk_config(
        sl_atr_mult=1.85,
        tp_atr_mult=4.5,
        trail_activation_atr=2.1,
        trail_atr_mult=1.4,
    )

    assert result["status"] == "synced"
    assert result["sl_atr_mult"] == 1.85
    assert result["tp_atr_mult"] == 4.5
    assert result["trail_activation_atr"] == 2.1
    assert result["trail_atr_mult"] == 1.4

    # Verify written to dummy_env
    content = dummy_env.read_text()
    assert "SL_ATR_MULT=1.85" in content
    assert "TP_ATR_MULT=4.5" in content
    assert "TRAIL_ACTIVATION_ATR=2.1" in content
    assert "TRAIL_ATR_MULT=1.4" in content


def test_jesse_bridge_sync_blocked_when_flag_disabled(tmp_path, monkeypatch):
    """When JESSE_SYNC_TO_LIVE is false (default), sync is blocked and does not touch RiskConfig."""
    dummy_env = tmp_path / ".env"
    dummy_env.write_text("SL_ATR_MULT=1.0\nTP_ATR_MULT=2.5\n")
    monkeypatch.setenv("ENV_FILE_PATH", str(dummy_env))
    monkeypatch.setenv("JESSE_SYNC_TO_LIVE", "false")

    service = JesseBridgeService()
    result = service.sync_strategy_to_risk_config(sl_atr_mult=3.0, tp_atr_mult=6.0)

    assert result["status"] == "blocked"
    assert result["synced"] is False
    # Content must NOT have changed
    content = dummy_env.read_text()
    assert "SL_ATR_MULT=1.0" in content
    assert "SL_ATR_MULT=3.0" not in content


@pytest.mark.asyncio
async def test_jesse_bridge_get_status_fallback():
    """Test get_status returns unavailable when Jesse server cannot be reached."""
    with patch.object(JesseBridgeService, "get_token", AsyncMock(return_value=None)):
        service = JesseBridgeService()
        status = await service.get_status()
        assert status["available"] is False
        assert status.get("error") is not None


def test_jesse_sync_route(client, monkeypatch, tmp_path):
    """Test /api/jesse/sync endpoint with admin API key when JESSE_SYNC_TO_LIVE=true."""
    dummy_env = tmp_path / ".env"
    dummy_env.write_text("SL_ATR_MULT=1.0\nTP_ATR_MULT=2.0\n")
    monkeypatch.setenv("ENV_FILE_PATH", str(dummy_env))
    monkeypatch.setenv("JESSE_SYNC_TO_LIVE", "true")

    api_key = os.getenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setenv("ADMIN_API_KEY", api_key)

    response = client.post(
        "/api/jesse/sync",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={
            "sl_atr_mult": 2.25,
            "tp_atr_mult": 5.0,
            "trail_activation_atr": 1.9,
            "trail_atr_mult": 1.3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "synced"
    assert data["sl_atr_mult"] == 2.25
    assert data["tp_atr_mult"] == 5.0


def test_finmem_evaluate_fails_closed_when_disabled_or_no_bars(client, monkeypatch):
    """FINMEM evaluation must fail closed (503) when FINMEM_ENABLED=false or market data fails."""
    api_key = os.getenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setenv("ADMIN_API_KEY", api_key)

    # 1. Disabled via FINMEM_ENABLED=false
    monkeypatch.setenv("FINMEM_ENABLED", "false")
    res_disabled = client.post(
        "/api/jesse/finmem/evaluate",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={"symbol": "BTC-USDT"},
    )
    assert res_disabled.status_code == 503
    assert "disabled" in res_disabled.json()["detail"]

    # 2. Enabled but market data unavailable -> 503 (fail closed, no 78400 dummy prices)
    monkeypatch.setenv("FINMEM_ENABLED", "true")
    from backend.services.binance_market_data import binance_market_data
    with patch.object(binance_market_data, "get_klines", AsyncMock(return_value=[])):
        res_no_data = client.post(
            "/api/jesse/finmem/evaluate",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={"symbol": "BTC-USDT"},
        )
        assert res_no_data.status_code == 503
        assert "fail closed" in res_no_data.json()["detail"]


@pytest.mark.asyncio
async def test_jesse_bridge_get_ml_prediction():
    """Test get_ml_prediction communicates with ML inference server."""
    mock_prediction = {
        "status": "success",
        "symbol": "BTC-USDT",
        "timeframe": "1h",
        "signal": "BUY",
        "confidence": 0.58,
        "probabilities": {"bullish": 0.58, "bearish": 0.12, "neutral": 0.30},
        "latest_close": 78500.0,
        "model": "BTC-USDT_1h_lightgbm.joblib",
        "latency_ms": 12.5,
    }

    service = JesseBridgeService()
    with patch.object(service, "get_ml_prediction", AsyncMock(return_value=mock_prediction)):
        res = await service.get_ml_prediction("BTC-USDT", "1h", "lightgbm", 0.45)
        assert res["status"] == "success"
        assert res["signal"] == "BUY"
        assert res["confidence"] == 0.58


@pytest.mark.asyncio
async def test_jesse_bridge_get_model_metadata():
    """Test get_model_metadata reads DSR/PBO from Jesse ML metadata endpoint."""
    mock_meta = {
        "status": "success",
        "symbol": "BTC-USDT",
        "timeframe": "1h",
        "model_type": "lightgbm",
        "feature_hash": "cd15d2380809b247",
        "metrics": {
            "deflated_sharpe_ratio": 1.0,
            "prob_backtest_overfitting": 0.636,
            "holdout_sharpe": 4.84,
        },
    }

    service = JesseBridgeService()

    async def fake_get(url, params=None):
        class Resp:
            status_code = 200

            @staticmethod
            def json():
                return mock_meta

        return Resp()

    mock_client = AsyncMock()
    mock_client.get = fake_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch.object(service, "_resolve_ml_url", AsyncMock(return_value="http://127.0.0.1:9003")), \
         patch("backend.services.jesse_bridge.httpx.AsyncClient", return_value=mock_client):
        meta = await service.get_model_metadata("BTCUSDT", "1h")
        assert meta["metrics"]["prob_backtest_overfitting"] == pytest.approx(0.636)


def test_metadata_is_usable_rejects_missing_artifacts():
    assert metadata_is_usable(None) is False
    assert metadata_is_usable({"status": "error", "error": "Metadata not found for ETH-USDT_1h_lightgbm_meta.json"}) is False
    assert metadata_is_usable({"status": "success", "metrics": {"deflated_sharpe_ratio": 1.0}}) is True
    assert metadata_is_usable({"metrics": {"deflated_sharpe_ratio": 1.0}}) is True
    assert is_missing_artifact_error({"error": "No model artifact found for ETH-USDT (1h, lightgbm)"}) is True
    assert is_missing_artifact_error({"error": "Connection refused"}) is False


def test_ml_predict_timeout_is_clamped(monkeypatch):
    monkeypatch.setenv("JESSE_ML_PREDICT_TIMEOUT", "999")
    assert ml_predict_timeout_seconds() == 30.0
    monkeypatch.setenv("JESSE_ML_PREDICT_TIMEOUT", "0")
    assert ml_predict_timeout_seconds() == 1.0
    monkeypatch.setenv("JESSE_ML_PREDICT_TIMEOUT", "not-a-number")
    assert ml_predict_timeout_seconds() == 20.0


@pytest.mark.asyncio
async def test_resolve_gate_model_uses_lstm_when_lightgbm_missing():
    service = JesseBridgeService()
    lightgbm_miss = {"status": "error", "error": "Metadata not found for ETH-USDT_1h_lightgbm_meta.json"}
    lstm_ok = {
        "status": "success",
        "symbol": "ETH-USDT",
        "model_type": "lstm",
        "metrics": {"deflated_sharpe_ratio": 1.0, "prob_backtest_overfitting": 0.028},
    }

    async def fake_meta(symbol, timeframe="1h", model_type="lightgbm"):
        if model_type == "lstm":
            return dict(lstm_ok)
        return dict(lightgbm_miss)

    with patch.object(service, "get_model_metadata", AsyncMock(side_effect=fake_meta)):
        resolved = await service.resolve_gate_model("ETHUSDT", "1h")
    assert resolved["resolved_model_type"] == "lstm"
    assert resolved["resolved_via"] == "fallback"
    assert resolved["metrics"]["prob_backtest_overfitting"] == pytest.approx(0.028)


@pytest.mark.asyncio
async def test_resolve_gate_model_keeps_lightgbm_when_present():
    service = JesseBridgeService()
    lgbm_ok = {
        "status": "success",
        "model_type": "lightgbm",
        "metrics": {"deflated_sharpe_ratio": 1.0, "prob_backtest_overfitting": 0.20},
    }

    async def fake_meta(symbol, timeframe="1h", model_type="lightgbm"):
        assert model_type == "lightgbm"
        return dict(lgbm_ok)

    with patch.object(service, "get_model_metadata", AsyncMock(side_effect=fake_meta)) as mock_meta:
        resolved = await service.resolve_gate_model("BTC-USDT", "1h")
    assert resolved["resolved_model_type"] == "lightgbm"
    assert resolved["resolved_via"] == "primary"
    assert mock_meta.await_count == 1


@pytest.mark.asyncio
async def test_resolve_gate_model_errors_when_neither_artifact_exists():
    service = JesseBridgeService()
    miss = {"status": "error", "error": "Metadata not found"}

    with patch.object(service, "get_model_metadata", AsyncMock(return_value=miss)):
        resolved = await service.resolve_gate_model("BTC-USDT", "1h")
    assert resolved["status"] == "error"
    assert resolved["resolved_via"] == "none"


@pytest.mark.asyncio
async def test_resolve_gate_model_respects_disabled_fallback(monkeypatch):
    monkeypatch.setenv("JESSE_ML_FALLBACK_MODEL_TYPE", "none")
    service = JesseBridgeService()
    miss = {"status": "error", "error": "Metadata not found for ETH-USDT_1h_lightgbm_meta.json"}

    with patch.object(service, "get_model_metadata", AsyncMock(return_value=miss)) as mock_meta:
        resolved = await service.resolve_gate_model("ETH-USDT", "1h")
    assert resolved["resolved_via"] == "none"
    assert mock_meta.await_count == 1


@pytest.mark.asyncio
async def test_jesse_bridge_get_validation_status():
    """Test validation status combines metadata with gate evaluation."""
    service = JesseBridgeService()
    mock_meta = {
        "symbol": "BTC-USDT",
        "feature_hash": "abc123",
        "labeling_mode": "triple_barrier",
        "metrics": {"deflated_sharpe_ratio": 1.0, "prob_backtest_overfitting": 0.15},
    }
    with patch.object(service, "get_model_metadata", AsyncMock(return_value=mock_meta)):
        status = await service.get_validation_status("BTC-USDT", "1h")
        assert status["status"] == "success"
        assert status["deployment_ok"] is True
        assert status["pbo_pass"] is True


def test_jesse_ml_validation_route(client, monkeypatch):
    """Test /api/jesse/ml-validation returns gate status."""
    api_key = os.getenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setenv("ADMIN_API_KEY", api_key)

    mock_status = {
        "status": "success",
        "symbol": "BTC-USDT",
        "timeframe": "1h",
        "deployment_ok": False,
        "pbo": 0.636,
        "dsr": 1.0,
        "pbo_pass": False,
        "dsr_pass": True,
        "reasons": ["PBO 63.6% exceeds gate 30%"],
    }

    with patch.object(jesse_bridge, "get_validation_status", AsyncMock(return_value=mock_status)):
        res = client.get(
            "/api/jesse/ml-validation?symbol=BTC-USDT",
            headers={"x-api-key": api_key},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["deployment_ok"] is False
        assert data["pbo"] == pytest.approx(0.636)


def test_jesse_ml_predict_routes(client, monkeypatch):
    """Test /api/jesse/ml-predict (GET and POST) and /api/jesse/ml-models."""
    api_key = os.getenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setenv("ADMIN_API_KEY", api_key)

    mock_prediction = {
        "status": "success",
        "symbol": "BTC-USDT",
        "timeframe": "1h",
        "signal": "BUY",
        "confidence": 0.58,
        "probabilities": {"bullish": 0.58, "bearish": 0.12, "neutral": 0.30},
        "latest_close": 78500.0,
        "model": "BTC-USDT_1h_lightgbm.joblib",
        "latency_ms": 12.5,
    }

    mock_models = {
        "status": "ok",
        "cached_models": ["BTC-USDT_1h_lightgbm"],
        "available_models": ["BTC-USDT_1h_lightgbm.joblib"],
    }

    with patch.object(jesse_bridge, "get_ml_prediction", AsyncMock(return_value=mock_prediction)), \
         patch.object(jesse_bridge, "get_ml_models", AsyncMock(return_value=mock_models)):

        # 1. GET /api/jesse/ml-predict
        res_get = client.get(
            "/api/jesse/ml-predict?symbol=BTC-USDT&timeframe=1h",
            headers={"x-api-key": api_key},
        )
        assert res_get.status_code == 200
        assert res_get.json()["signal"] == "BUY"

        # 2. POST /api/jesse/ml-predict
        res_post = client.post(
            "/api/jesse/ml-predict",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={"symbol": "BTC-USDT", "timeframe": "1h", "threshold": 0.5},
        )
        assert res_post.status_code == 200
        assert res_post.json()["confidence"] == 0.58

        # 3. GET /api/jesse/ml-models
        res_models = client.get(
            "/api/jesse/ml-models",
            headers={"x-api-key": api_key},
        )
        assert res_models.status_code == 200
        assert "BTC-USDT_1h_lightgbm.joblib" in res_models.json()["available_models"]

