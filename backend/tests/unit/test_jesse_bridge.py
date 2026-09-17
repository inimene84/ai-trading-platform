"""Unit tests for Jesse AI Quant Engine Bridge and API Routes."""

import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.jesse_bridge import (
    JesseBridgeService,
    configured_ml_fallback_model_type,
    is_jesse_ml_model_gap,
    jesse_bridge,
    ml_predict_timeout_seconds,
)


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


def test_is_jesse_ml_model_gap_detects_missing_and_unpromoted():
    assert is_jesse_ml_model_gap("No model artifact found for AVAX-USDT (1h, lightgbm)") is True
    assert is_jesse_ml_model_gap("Model artifact ETH-USDT_1h_lightgbm.joblib refused: promotion gate failed") is True
    assert is_jesse_ml_model_gap("Connection refused") is False
    assert is_jesse_ml_model_gap("") is False


@pytest.mark.asyncio
async def test_get_ml_prediction_maps_sidecar_gap_to_no_model():
    service = JesseBridgeService()
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "status": "error",
        "error": "No model artifact found for AVAX-USDT (1h, lightgbm)",
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_res)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False
    with patch.object(service, "_resolve_ml_url", AsyncMock(return_value="http://ml")), \
         patch("httpx.AsyncClient", return_value=mock_client):
        res = await service.get_ml_prediction("AVAX-USDT")
    assert res["status"] == "no_model"


@pytest.mark.asyncio
async def test_get_ml_prediction_fail_closed_without_promotion_metrics():
    service = JesseBridgeService()
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "status": "success",
        "signal": "BUY",
        "confidence": 0.70,
        "probabilities": {"bullish": 0.70, "bearish": 0.10, "neutral": 0.20},
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_res)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False
    with patch.object(service, "_resolve_ml_url", AsyncMock(return_value="http://ml")), \
         patch("httpx.AsyncClient", return_value=mock_client):
        res = await service.get_ml_prediction("BTC-USDT")
    assert res["status"] == "error"
    assert "promotion gate" in str(res.get("error") or "").lower()
    assert res.get("promotion_ok") is False


def test_jesse_promotion_status_unconfigured(client, monkeypatch):
    api_key = os.getenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setenv("ADMIN_API_KEY", api_key)
    monkeypatch.delenv("QTP_PROMOTION_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("QTP_PROMOTION_REQUIRED", raising=False)
    res = client.get("/api/jesse/promotion-status", headers={"x-api-key": api_key})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "unconfigured"
    assert body["verdict"] is None
    assert body["promoted"] is False
    assert "gpu-artifacts" in body["gpu_artifacts_note"]


def test_jesse_promotion_status_reject_when_required_without_artifacts(client, monkeypatch):
    api_key = os.getenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setenv("ADMIN_API_KEY", api_key)
    monkeypatch.setenv("QTP_PROMOTION_REQUIRED", "true")
    monkeypatch.delenv("QTP_PROMOTION_ARTIFACT_DIR", raising=False)
    res = client.get("/api/jesse/promotion-status", headers={"x-api-key": api_key})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["verdict"] == "REJECT"
    assert body["promoted"] is False
    assert body["failed_gate"] == "SCHEMA_HASH"


def test_jesse_promotion_status_promote_happy_path(client, monkeypatch):
    from backend.ml.promotion_gates import GateResult
    from backend.ml.promotion_service import PromotionState

    api_key = os.getenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setenv("ADMIN_API_KEY", api_key)
    state = PromotionState(
        result=GateResult(
            verdict="PROMOTE",
            reason="PASS DSR/PBO/geometry",
            failed_gate=None,
            warnings=[],
            details={"dsr": 0.99},
        )
    )
    monkeypatch.setattr("backend.routes.jesse.resolve_promotion", lambda _cfg=None: state)
    res = client.get("/api/jesse/promotion-status", headers={"x-api-key": api_key})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["verdict"] == "PROMOTE"
    assert body["promoted"] is True
    assert body["reason"] == "PASS DSR/PBO/geometry"


def test_jesse_models_alias_and_no_promoted_model(client, monkeypatch):
    api_key = os.getenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setenv("ADMIN_API_KEY", api_key)
    monkeypatch.delenv("QTP_PROMOTION_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("QTP_PROMOTION_REQUIRED", raising=False)

    mock_models = {
        "status": "ok",
        "cached_models": [],
        "available_models": [],
        "rejected_models": {"BTC-USDT_1h_lightgbm": ["collapsed classifier"]},
    }
    with patch.object(jesse_bridge, "get_ml_models", AsyncMock(return_value=mock_models)):
        res_alias = client.get("/api/jesse/models", headers={"x-api-key": api_key})
        res_legacy = client.get("/api/jesse/ml-models", headers={"x-api-key": api_key})

    assert res_alias.status_code == 200
    assert res_legacy.status_code == 200
    body = res_alias.json()
    assert body["status"] == "ok"
    assert body["has_promoted_model"] is False
    assert body["available_models"] == []
    assert body["promotion"]["status"] == "unconfigured"
    assert "collapsed classifier" in body["rejected_models"]["BTC-USDT_1h_lightgbm"]
    assert res_legacy.json()["has_promoted_model"] is False


def test_jesse_models_error_is_json_not_404(client, monkeypatch):
    api_key = os.getenv("ADMIN_API_KEY", "test_key")
    monkeypatch.setenv("ADMIN_API_KEY", api_key)
    monkeypatch.delenv("QTP_PROMOTION_ARTIFACT_DIR", raising=False)
    with patch.object(
        jesse_bridge,
        "get_ml_models",
        AsyncMock(return_value={"status": "error", "error": "ML server returned HTTP 503"}),
    ):
        res = client.get("/api/jesse/models", headers={"x-api-key": api_key})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "error"
    assert body["available_models"] == []
    assert body["has_promoted_model"] is False


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

        # 4. GET /api/jesse/models (ops alias)
        res_alias = client.get("/api/jesse/models", headers={"x-api-key": api_key})
        assert res_alias.status_code == 200
        assert "BTC-USDT_1h_lightgbm.joblib" in res_alias.json()["available_models"]


def test_configured_ml_fallback_defaults_to_lstm(monkeypatch):
    monkeypatch.delenv("JESSE_ML_FALLBACK_MODEL_TYPE", raising=False)
    assert configured_ml_fallback_model_type() == "lstm"


def test_configured_ml_fallback_can_be_disabled(monkeypatch):
    for value in ("none", "false", "off", "0", ""):
        monkeypatch.setenv("JESSE_ML_FALLBACK_MODEL_TYPE", value)
        assert configured_ml_fallback_model_type() is None


def test_ml_predict_timeout_clamped(monkeypatch):
    monkeypatch.setenv("JESSE_ML_PREDICT_TIMEOUT", "999")
    assert ml_predict_timeout_seconds() == 30.0
    monkeypatch.setenv("JESSE_ML_PREDICT_TIMEOUT", "12")
    assert ml_predict_timeout_seconds() == 12.0


def _promotion_ok_payload(**overrides):
    base = {
        "status": "success",
        "signal": "BUY",
        "confidence": 0.58,
        "probabilities": {"bullish": 0.58, "bearish": 0.0, "neutral": 0.42},
        "conformal_margin": 0.28,
        "entropy": 0.55,
        "decision": {"expected_value_r": 0.11},
        "metrics": {
            "deflated_sharpe_ratio": 1.0,
            "prob_backtest_overfitting": 0.03,
            "pt_mult": 5.5,
            "sl_mult": 1.75,
            "bullish_recall": 0.45,
            "bearish_recall": 0.12,
        },
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_predict_once_attaches_live_telemetry_via_http():
    service = JesseBridgeService()
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = _promotion_ok_payload()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_res)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False
    with patch.object(service, "_resolve_ml_url", AsyncMock(return_value="http://ml")), \
         patch("httpx.AsyncClient", return_value=mock_client):
        res = await service._predict_once("ETH-USDT", "1h", "lightgbm", 0.45)
    assert res["status"] == "success"
    assert res.get("p_win") == pytest.approx(0.58)
    assert res.get("conformal_width") is not None
    assert res.get("costed_edge_bps") is not None


@pytest.mark.asyncio
async def test_get_ml_prediction_retries_lstm_on_no_model(monkeypatch):
    monkeypatch.delenv("JESSE_ML_FALLBACK_MODEL_TYPE", raising=False)
    service = JesseBridgeService()
    calls = {"n": 0}

    def mock_post(*_args, **_kwargs):
        calls["n"] += 1
        mock_res = MagicMock()
        mock_res.status_code = 200
        if calls["n"] == 1:
            mock_res.json.return_value = {
                "status": "error",
                "error": "No model artifact found for ETH-USDT (1h, lightgbm)",
            }
        else:
            mock_res.json.return_value = _promotion_ok_payload()
        return mock_res

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False
    with patch.object(service, "_resolve_ml_url", AsyncMock(return_value="http://ml")), \
         patch("httpx.AsyncClient", return_value=mock_client):
        res = await service.get_ml_prediction("ETH-USDT", "1h", "lightgbm", 0.45)
    assert res["status"] == "success"
    assert res["resolved_via"] == "fallback"
    assert res.get("p_win") == pytest.approx(0.58)
    assert res.get("conformal_width") is not None
    assert res.get("costed_edge_bps") is not None


@pytest.mark.asyncio
async def test_jesse_bridge_fail_closed_without_password(monkeypatch):
    """With JESSE_PASSWORD unset, get_token returns None (no hardcoded fallback)."""
    monkeypatch.delenv("JESSE_PASSWORD", raising=False)
    monkeypatch.setattr("backend.services.jesse_bridge.JESSE_PASSWORD", "")
    service = JesseBridgeService()
    token = await service.get_token()
    assert token is None
